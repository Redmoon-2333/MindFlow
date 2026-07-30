"""Graph state contracts, reducers, routing, and panel graph for MindFlow LangGraph orchestration.

Exports:
    State classes: ``AnalysisState``, ``PanelState``, ``ChatState``
    Reducers: ``append_opinion``, ``append_transcript``, ``accumulate_tool_messages``,
        ``accumulate_errors``
    Routes: ``AnalysisRoute``, ``PanelRoute``, ``ChatRoute``

Note: ``PanelGraph`` lives in ``mindflow.graph.panel_graph`` and is
imported from there directly (not re-exported here) to avoid a
circular import with ``mindflow.agents.orchestrator``.
"""

from mindflow.graph.reducers import (
    accumulate_errors,
    accumulate_tool_messages,
    append_opinion,
    append_transcript,
)
from mindflow.graph.routing import AnalysisRoute, ChatRoute, PanelRoute
from mindflow.graph.state import AnalysisState, ChatState, PanelState

__all__ = [
    "AnalysisState",
    "PanelState",
    "ChatState",
    "append_opinion",
    "append_transcript",
    "accumulate_tool_messages",
    "accumulate_errors",
    "AnalysisRoute",
    "PanelRoute",
    "ChatRoute",
]
