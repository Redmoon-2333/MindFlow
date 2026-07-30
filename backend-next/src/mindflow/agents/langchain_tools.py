"""MindFlow tools declared as LangChain ``@tool`` for use with ``create_agent``.

Each tool wraps a typed adapter from ``mindflow.graph.tools`` and returns a
string suitable for inclusion in the LLM context window.  Adapters receive
dependencies via constructor injection and context via an explicit
``ToolContext`` set before agent invocation — no ContextVars, no global state.

Per-session caps:
  - ``run_panel``: 1 invocation per session (enforced by
    ``RunAnalysisTool`` via ``BudgetReservationPort``).
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from mindflow.graph.tools import (
    InterventionHistoryTool,
    LatestAnalysisTool,
    QueryEvidenceTool,
    RunAnalysisTool,
    format_analysis_output,
    format_evidence_output,
    format_intervention_output,
    format_run_output,
)

# ── Tool factory: query_evidence ─────────────────────────────────────────


def make_query_evidence(
    adapter: QueryEvidenceTool,
) -> BaseTool:
    """Return a ``query_evidence`` tool bound to *adapter*.

    The tool signature exposed to the LLM::

        query_evidence(days_back: int = 7) -> str
    """

    @tool
    async def query_evidence(days_back: int = 7) -> str:
        """Query behavior evidence from the ML sensing layer.

        Fetches focus score, switch rate, longest focus block, behavior
        deviation, intervention history, and novelty flags for the last
        N days (capped at 30).

        Args:
            days_back: Number of days to look back (max 30).

        Returns:
            JSON string of the evidence bundle.
        """
        result = await adapter.execute(days=days_back)
        return format_evidence_output(result)

    return query_evidence


# ── Tool factory: get_latest_analysis ────────────────────────────────────


def make_get_latest_analysis(
    adapter: LatestAnalysisTool,
) -> BaseTool:
    """Return a ``get_latest_analysis`` tool bound to *adapter*.

    The tool signature exposed to the LLM::

        get_latest_analysis() -> str
    """

    @tool
    async def get_latest_analysis() -> str:
        """Retrieve today's (or yesterday's) procrastination analysis.

        Returns the latest procrastination-type diagnosis with confidence
        scores from the ML pipeline.

        Returns:
            JSON string of the analysis result, or a not-found message.
        """
        result = await adapter.execute()
        return format_analysis_output(result)

    return get_latest_analysis


# ── Tool factory: run_panel ──────────────────────────────────────────────


def make_run_panel(
    adapter: RunAnalysisTool,
) -> BaseTool:
    """Return a ``run_panel`` tool bound to *adapter*.

    The tool signature exposed to the LLM::

        run_panel() -> str

    Per-session cap (1 call) is enforced by the adapter via
    ``BudgetReservationPort``.
    """

    @tool
    async def run_panel() -> str:
        """Run the expert panel deliberation on today's data.

        Triggers a multi-expert analysis (analyst, attribution expert,
        moderator, critic) to produce a procrastination-type verdict
        with CBT recommendations.

        Limited to **1 invocation per session**.

        Returns:
            JSON string of the panel verdict, or an error/skip message.
        """
        from mindflow.time_utils import business_today  # noqa: PLC0415

        target_date = business_today("local")
        result = await adapter.execute(date=target_date)
        return format_run_output(result)

    return run_panel


# ── Tool factory: query_interventions ────────────────────────────────────


def make_query_interventions(
    adapter: InterventionHistoryTool,
) -> BaseTool:
    """Return a ``query_interventions`` tool bound to *adapter*.

    The tool signature exposed to the LLM::

        query_interventions(days_back: int = 7) -> str
    """

    @tool
    async def query_interventions(days_back: int = 7) -> str:
        """Query recent intervention history.

        Returns nudge, task-breakdown, reframe, and environment-mod
        intervention records triggered in the last N days (capped at 30).

        Args:
            days_back: Number of days to look back (max 30).

        Returns:
            JSON string of intervention records, or a not-found message.
        """
        result = await adapter.execute(days=days_back)
        return format_intervention_output(result)

    return query_interventions
