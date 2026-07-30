"""Route contracts for MindFlow LangGraph state machines.

Defines Literal-type enums that nodes return to the graph engine
so edges can be evaluated deterministically without runtime imports
from the orchestrator module.

All route types are framework-neutral — they carry no LangGraph dependency.
"""

from __future__ import annotations

from typing import Literal

# ── Analysis orchestration levels (degradation chain) ───────────────────────────

AnalysisRoute = Literal[
    "panel",            # Multi-expert panel deliberation
    "single_expert",    # Fallback: single LLM expert
    "ollama",           # Fallback: local Ollama model
    "rule_engine",      # Last resort: deterministic rules
    "end",              # Terminal: verdict produced or unrecoverable
]

"""Valid routing targets within the top-level analysis graph.

Mirrors ``PanelSource`` in ``mindflow.agents.types`` but adds ``"end"``
as a terminal sentinel for graph edge evaluation.
"""

# ── Panel deliberation phases ───────────────────────────────────────────────────

PanelRoute = Literal[
    "analyst",          # Round 0: data analyst opinion
    "attribution",      # Round 1: parallel attribution expert fan-out
    "conflict_check",   # Gate: detect disagreement → escalate or skip
    "rebuttal",         # Round 2a (optional): rebuttal round
    "moderator",        # Round 2/3: moderator synthesizes verdict
    "human_review",     # Gate (optional): human review interrupt before critic
    "critic",           # Round 3/4: critic validates verdict
    "critic_retry",     # Gate: critic rejected → loop back to moderator
    "end",              # Terminal
]

"""Valid routing targets within the panel deliberation sub-graph.

Order follows the fast-path flow (§4 in 07-agent-upgrade-design.md):
analyst → attribution×3 → conflict_check → moderator → critic → end.
"""

# ── Moderator and Critic route outcomes (Todo 10) ────────────────────────────────

ModeratorRoute = Literal[
    "approved",         # Verdict accepted by critic
    "retry",            # Critic rejected — loop back for one redo
    "exhausted",        # Retries exceeded — end with last verdict
]

"""Moderator routing outcomes after critic validation.

``critic_retries`` state field tracks how many times the moderator re-verdict
loop has been entered:
  - First moderator pass → critic → approved      → END
  - First moderator pass → critic → rejected      → ``retry`` (moderator redo)
  - Moderator redo      → critic → approved      → END
  - Moderator redo      → critic → rejected      → ``exhausted`` → END
"""

CriticRoute = Literal[
    "approved",         # Verdict passes all checks
    "rejected",         # Verdict has issues — triggers retry or exhaustion
]

"""Critic outcome after validating a moderator verdict.

When ``rejected``, the moderator may retry once (``retry``) or exhaust
(``exhausted``) depending on ``critic_retries`` count.
"""

# ── Chat conversation phases ────────────────────────────────────────────────────

ChatRoute = Literal[
    "crisis_check",     # Pre-LLM crisis detection gate
    "agent_invoke",     # LangChain agent with tool loop
    "forbidden_check",  # Post-LLM forbidden word scan
    "retry",            # Retry loop (forbidden word / tool error)
    "end",              # Terminal
]

"""Valid routing targets within the chat conversation graph.

Corresponds to the pipeline steps in ``ChatService.ask()``
(chat_service.py §1-6).
"""
