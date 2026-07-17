"""Adapters that wrap analyzers into the eval ``run_eval`` interface.

Provides two adapter factories:
  - ``rule_engine_analyzer`` — wraps the deterministic RuleEngine (L3).
  - ``panel_analyzer`` — wraps the LLM expert panel (L1), accepting either
    a real ``DeepSeekGateway`` or a ``MockPanelGateway`` for pipeline verification.

The mock gateway implements a simplified rule-based heuristic that produces
plausible panel responses from bundle features. **It is NOT a measure of
real expert-panel capability** — its purpose is pipeline verification and
to exercise the evaluation metrics with a known response distribution.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from mindflow.agents.llm_gateway import PanelLLMGateway
from mindflow.agents.orchestrator import PanelOrchestrator
from mindflow.agents.types import PanelVerdict
from mindflow.domain.evidence import EvidenceBundle
from mindflow.domain.procrastination import (
    ProcrastinationAssessment,
    RuleEngine,
)

# ---------------------------------------------------------------------------
# Rule engine adapter
# ---------------------------------------------------------------------------


async def rule_engine_analyzer(bundle: EvidenceBundle) -> ProcrastinationAssessment:
    """Run the deterministic rule engine on a bundle.

    Requires ``bundle.behavior_summary`` to be present.

    Raises:
        ValueError: If the bundle has no behavior_summary.
    """
    summary = bundle.behavior_summary
    if summary is None:
        raise ValueError("Bundle has no behavior_summary — cannot run rule engine")
    return RuleEngine().assess(summary)


# ---------------------------------------------------------------------------
# Panel analyzer (factory)
# ---------------------------------------------------------------------------

AnalyzerFunc = Callable[[EvidenceBundle], Awaitable[ProcrastinationAssessment | PanelVerdict]]


def panel_analyzer(gateway: PanelLLMGateway) -> AnalyzerFunc:
    """Create an analyzer that runs the expert panel via the given gateway.

    Args:
        gateway: A PanelLLMGateway implementation (real DeepSeekGateway or mock).

    Returns:
        An async callable that takes an EvidenceBundle and returns a PanelVerdict.
    """

    async def _analyze(bundle: EvidenceBundle) -> PanelVerdict:
        orchestrator = PanelOrchestrator(gateway=gateway)
        return await orchestrator.run(bundle)

    return _analyze


# ===================================================================
# MockPanelGateway — scripted mock for pipeline verification
# ===================================================================
#
# *** Pipeline verification only — NOT a substitute for real model eval ***
#
# The mock inspects bundle fields from the user prompt (which contains the
# serialized ``to_prompt_json`` output) and applies simplified threshold rules
# to produce scenario-appropriate responses for each expert role.
#
# This exercises the evaluation pipeline end-to-end with a known, deterministic
# response distribution, allowing us to verify that metrics (Top-1 accuracy,
# Jaccard, technique match) are computed correctly.
#
# Response distribution (designed for 30 eval scenarios):
#   - ~80% correct Top-1 (matches gold standard)
#   - ~13% wrong Top-1 (plausible but incorrect)
#   - ~7% partial (correct type but wrong ordering)
# ===================================================================

# Expert role fingerprints (matching orchestrator system prompts)
_FP_ANALYST = "行为数据分析师"
_FP_CBT = "认知行为疗法"
_FP_TMT = "时间动机理论"
_FP_EMOTION = "情绪调节归因专家"
_FP_MODERATOR = "会诊综合主持人"
_FP_CRITIC = "批评家"

# Technique name mapping
_TECHNIQUE_MAP: dict[str, str] = {
    "task_aversion": "graded_exposure",
    "impulsivity": "stimulus_control",
    "decisional": "goal_setting",
    "perfectionism": "cognitive_restructuring",
    "emotional_regulation": "mindfulness",
}

# Metrics that are guaranteed to exist in every bundle's evidence items.
# The orchestrator's citation validator checks against metric_names(bundle),
# so mock citations must only reference metrics that actually appear.
_SAFE_METRICS: frozenset[str] = frozenset({
    "focus_score", "switch_rate", "longest_focus_block_s",
    "social_media_ratio",
})


def _classify_from_metrics(metrics: dict[str, float]) -> tuple[list[str], list[float], str]:
    """Simplified classification based on bundle metrics.

    Returns (type_names, confidences, top_technique).
    """
    types: list[str] = []
    confs: list[float] = []
    switches = metrics.get("context_switches_per_hour", 0)
    block = metrics.get("longest_focus_block_sec", metrics.get("longest_focus_block_s", 999))
    delay = metrics.get("start_delay_min", 0)
    social = metrics.get("social_media_ratio", 0)
    focus_ratio = metrics.get("focus_ratio", 0.5)
    deviation = metrics.get("baseline_deviation", 0)

    # Impulsivity: high switches AND short blocks
    if switches >= 12 and block < 300:
        imp_conf = min(0.95, 0.5 + (switches - 12) / 12 * 0.45)
        types.append("impulsivity")
        confs.append(imp_conf)

    # Decisional: significant delay with good focus recovery
    if delay > 30 and focus_ratio > 0.4:
        dec_conf = min(0.95, 0.5 + (delay - 30) / 30 * 0.45)
        types.append("decisional")
        confs.append(dec_conf)

    # Emotional regulation: high social media ratio
    if social > 0.55:
        emo_conf = min(0.95, 0.5 + (social - 0.55) / 0.25 * 0.45)
        types.append("emotional_regulation")
        confs.append(emo_conf)

    # Task aversion (catch-all)
    if not types:
        if focus_ratio < 0.35 or deviation < -0.5:
            types.append("task_aversion")
            confs.append(0.6)
        else:
            types.append("impulsivity")
            confs.append(0.15)

    # Perfectionism — only detected if explicitly flagged in evidence items
    # (keyword_flags not serialized in to_prompt_json, so rarely matched here)
    if metrics.get("has_keyword_flags", 0) > 0:
        types.append("perfectionism")
        confs.append(0.6)

    # Determine top technique
    top_type = types[0] if types else "impulsivity"
    top_technique = _TECHNIQUE_MAP.get(top_type, "stimulus_control")

    return types, confs, top_technique


def _make_citations(types: list[str], available_metrics: frozenset[str] | None = None) -> list[str]:
    """Return plausible evidence citations for the detected types.

    Only includes metrics that exist in *available_metrics* (the bundle's
    evidence items), so the orchestrator's citation validator doesn't skip
    the mock's responses as hallucinated.
    """
    candidates = ["focus_score"]
    if "impulsivity" in types:
        candidates.extend(["switch_rate", "longest_focus_block_s"])
    if "emotional_regulation" in types:
        candidates.append("social_media_ratio")
    if "decisional" in types:
        candidates.append("start_delay_min")
    if "task_aversion" in types:
        candidates.append("focus_score")

    if available_metrics:
        candidates = [m for m in candidates if m in available_metrics]

    return list(dict.fromkeys(candidates))  # dedup preserving order


def _lookup_evidence_value(
    evidence: list[dict[str, Any]], metric: str,
) -> float | None:
    """Find a metric value in the evidence list."""
    for item in evidence:
        if item.get("metric") == metric:
            raw = item.get("value")
            if raw is not None:
                return float(raw)
    return None


def _extract_metrics_from_user(user: str) -> tuple[dict[str, float], frozenset[str]]:
    """Extract behavioral metrics + available metric names from user prompt JSON.

    The user prompt contains the output of ``to_prompt_json()`` + possible
    additional context from the orchestrator. This method extracts the serialized
    JSON and parses out key metrics, along with the set of metric names present
    in the evidence items (used for citation validation).

    Returns:
        (metrics_dict, available_metric_names)
    """
    metrics: dict[str, float] = {}
    available: set[str] = set()

    try:
        start_idx = -1
        for marker in ('{"window":', '{"evidence":'):
            idx = user.find(marker)
            if idx >= 0:
                start_idx = idx
                break

        if start_idx >= 0:
            depth = 0
            end_idx = -1
            in_str = False
            escape = False
            for i in range(start_idx, len(user)):
                ch = user[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break

            if end_idx > start_idx:
                raw_json = user[start_idx: end_idx + 1]
                data = json.loads(raw_json)

                # Extract from behavior_summary
                bs = data.get("behavior_summary", {})
                for key in (
                    "duration_min", "actual_focus_min",
                    "context_switches_per_hour", "longest_focus_block_sec",
                    "social_media_ratio", "start_delay_min",
                ):
                    if key in bs and bs[key] is not None:
                        metrics[key] = float(bs[key])

                if "baseline_deviation" in bs and bs["baseline_deviation"] is not None:
                    metrics["baseline_deviation"] = float(bs["baseline_deviation"])

                # Compute focus_ratio
                if "actual_focus_min" in metrics and "duration_min" in metrics:
                    d = metrics["duration_min"]
                    if d > 0:
                        metrics["focus_ratio"] = metrics["actual_focus_min"] / d

                # Extract from evidence items
                evidence = data.get("evidence", [])
                for ev in evidence:
                    metric_name = ev.get("metric", "")
                    value = ev.get("value")
                    if metric_name:
                        available.add(metric_name)
                    if metric_name and value is not None and metric_name not in metrics:
                        metrics[metric_name] = float(value)

                # Look for keyword indicators
                if "self_criticism" in user or "redo_pattern" in user:
                    metrics["has_keyword_flags"] = 1.0

    except (json.JSONDecodeError, ValueError, KeyError, IndexError):
        pass

    return metrics, frozenset(available)


class MockPanelGateway:
    """Rule-based mock gateway that produces plausible panel responses.

    *** Pipeline verification only — NOT a substitute for real model eval ***

    The mock inspects key behavioral metrics from the user prompt and applies
    simple threshold rules to produce responses that mimic what an expert panel
    might say. This exercises the evaluation pipeline end-to-end with a known,
    deterministic response distribution — allowing us to verify that the metrics
    (Top-1 accuracy, Jaccard, technique match) are computed correctly.

    Response distribution (designed for 30 eval scenarios):
      - ~80% correct Top-1 (matches gold standard)
      - ~13% wrong Top-1 (plausible but incorrect)
      - ~7% partial (correct type but wrong ordering)

    Usage:
        gateway = MockPanelGateway()
        analyzer = panel_analyzer(gateway)
        report = await run_eval(analyzer, ALL_SCENARIOS, analyzer_name="panel")
    """

    _counts: dict[str, int]
    _cached_metrics: dict[str, float]
    _available_metrics: frozenset[str]

    def __init__(self) -> None:
        self._counts = {}
        self._cached_metrics = {}
        self._available_metrics = frozenset()

    async def complete(
        self,
        system: str,
        user: str,
        model: Literal["chat", "reasoner"] = "chat",  # noqa: ARG002
    ) -> str:
        """Return a plausible JSON response based on expert role and bundle data.

        Inspects the user prompt for bundle features and applies simplified
        classification rules to produce a response appropriate to the expert role.
        """
        role = self._classify_role(system)

        # Extract metrics from user prompt (works for analyst/attribution calls
        # where user is the pure bundle JSON; for moderator/critic calls the
        # bundle JSON is embedded in a larger prompt).
        fresh_metrics, fresh_available = _extract_metrics_from_user(user)
        if fresh_metrics:
            self._cached_metrics = fresh_metrics
            self._available_metrics = fresh_available

        # Reset per-scenario tracking on new scenario (analyst is always first)
        if role == "analyst":
            self._counts.clear()
            self._cached_metrics = fresh_metrics
            self._available_metrics = fresh_available

        self._counts[role] = self._counts.get(role, 0) + 1
        call_n = self._counts[role]

        if role == "analyst":
            return self._analyst_response()
        elif role in ("cbt", "tmt", "emotion"):
            return self._attribution_response(role, call_n)
        elif role == "moderator":
            return self._moderator_response(call_n)
        elif role == "critic":
            return self._critic_response(call_n)
        else:
            return '{"approved": true, "issues": []}'

    async def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Role classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_role(system: str) -> str:
        if _FP_ANALYST in system:
            return "analyst"
        if _FP_CBT in system:
            return "cbt"
        if _FP_TMT in system:
            return "tmt"
        if _FP_EMOTION in system:
            return "emotion"
        if _FP_MODERATOR in system:
            return "moderator"
        if _FP_CRITIC in system:
            return "critic"
        return "unknown"

    # ------------------------------------------------------------------
    # Response generators
    # ------------------------------------------------------------------

    def _classify(self) -> tuple[list[str], list[float], str]:
        """Classify using cached metrics."""
        return _classify_from_metrics(self._cached_metrics)

    def _analyst_response(self) -> str:
        types, confs, _ = self._classify()
        citations = _make_citations(types, self._available_metrics)

        type_labels = {
            "impulsivity": "冲动分心模式",
            "decisional": "决策困难模式",
            "perfectionism": "完美主义倾向",
            "emotional_regulation": "情绪调节模式",
            "task_aversion": "任务畏惧模式",
        }

        patterns_list: list[dict[str, str]] = []
        for i, t in enumerate(types[:2]):
            label = type_labels.get(t, t)
            sev: str = "severe" if i == 0 and confs and (confs[i] if i < len(confs) else 0) > 0.8 else "moderate"
            patterns_list.append({
                "name": label, "severity": sev,
                "description": f"检测到{label}",
            })

        top_concerns = [type_labels.get(t, t) for t in types[:3]]

        return json.dumps({
            "patterns": patterns_list,
            "anomalies": [],
            "top_concerns": top_concerns,
            "evidence_citations": citations,
        }, ensure_ascii=False)

    def _attribution_response(self, role: str, call_n: int) -> str:
        types, confs, _ = self._classify()

        if not types:
            types = ["task_aversion"]
            confs = [0.5]

        # Each attribution expert picks one type
        type_name = types[0]
        conf_idx = types.index(type_name)
        att_conf = confs[conf_idx] if conf_idx < len(confs) else 0.5
        if call_n > 1:
            att_conf = max(0.3, att_conf - 0.05)

        citations = _make_citations(types, self._available_metrics)
        args_list = [f"[证据: {c}]" for c in citations]
        argument = (
            f"从{self._attribution_perspective(role)}角度分析，"
            f"用户表现出{type_name}拖延模式。{' '.join(args_list)}"
        )

        return json.dumps({
            "attribution_types": [type_name],
            "confidence": {type_name: round(att_conf, 2)},
            "argument": argument,
            "evidence_citations": citations,
        }, ensure_ascii=False)

    @staticmethod
    def _attribution_perspective(role: str) -> str:
        perspectives = {
            "cbt": "认知行为理论",
            "tmt": "时间动机理论",
            "emotion": "情绪调节理论",
        }
        return perspectives.get(role, "行为分析")

    def _moderator_response(self, call_n: int) -> str:  # noqa: ARG002
        types_list, confs_list, technique = self._classify()

        if not types_list:
            types_list = ["task_aversion"]
            confs_list = [0.5]

        confidence_dict = {t: round(c, 2) for t, c in zip(types_list, confs_list, strict=False)}

        return json.dumps({
            "types": types_list[:3],
            "confidence": confidence_dict,
            "recommended_technique": technique,
            "rationale": "综合多方专家意见后，得出上述评估结论。",
            "dissent": [],
        }, ensure_ascii=False)

    @staticmethod
    def _critic_response(call_n: int) -> str:
        return json.dumps({
            "approved": True,
            "issues": [],
            "critique_detail": "通过。" if call_n <= 1 else "重审通过。",
        })
