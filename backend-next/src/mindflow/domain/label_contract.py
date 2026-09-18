"""Label contract: keep three different questions separate.

The spec calls out — correctly — that these are three questions, not one:

1. **Was the user distracted then?**  → ``StateLabel``. Trains the state model.
2. **Did the reminder help?**         → ``HelpfulnessRating``. Evaluation of
   interventions.
3. **Was the reminder intrusive?**    → ``IntrusivenessRating``. Reminder policy.

Collapsing (2) or (3) into (1) would let "the reminder was annoying" masquerade
as "the user was focused". The existing intervention feedback endpoint already
carries (2)/(3) semantics under ``helpful``/``neutral``/``annoying``; this
module names them so downstream code cannot conflate them.

Also defines the optional task context. Critically:

* missing context is ``unknown``, **not** a default value;
* a deliberate break is ``break`` and is *not* rewritten into focus/distracted —
  it only protects the user from reminders;
* ``goal_alignment`` says whether the work matched the plan, which is a
  different question from whether the user was focused.
"""

from __future__ import annotations

from typing import Literal

# ── 1. State (trains the model) ──────────────────────────────────────────

StateLabel = Literal["focus", "distracted", "mixed"]

# ── 2/3. Intervention evaluation (never trains the state model) ──────────

HelpfulnessRating = Literal["helpful", "neutral", "annoying"]
IntrusivenessRating = Literal["helpful", "neutral", "annoying"]

# ── Optional task context (all default to unknown) ───────────────────────

ContextType = Literal[
    "task_execution",     # actively working on the task
    "task_lookup",        # looking something up for the task
    "break",              # deliberate rest
    "other",
    "unknown",
]

GoalAlignment = Literal["aligned", "deviated", "uncertain", "unknown"]

LabelSource = Literal[
    "user_confirmed",     # the user explicitly submitted this
    "window_annotation",  # auxiliary per-window calibration
    "rule_inferred",      # heuristic inference
    "llm_assisted",       # model-generated suggestion
    "unknown",            # historical row with no recorded provenance
]

CONTEXT_TYPES: tuple[str, ...] = (
    "task_execution", "task_lookup", "break", "other", "unknown",
)
GOAL_ALIGNMENTS: tuple[str, ...] = ("aligned", "deviated", "uncertain", "unknown")
LABEL_SOURCES: tuple[str, ...] = (
    "user_confirmed", "window_annotation", "rule_inferred", "llm_assisted", "unknown",
)

#: Task vocabulary — unchanged, reused rather than redefined so the existing
#: classifier and this contract cannot drift apart.
TASK_TYPES: tuple[str, ...] = (
    "coding", "writing", "study", "meeting", "admin", "creative", "other",
    "gaming", "entertainment", "browsing", "communication",
)

UNKNOWN = "unknown"

#: A `break` protects the user from interruption. It is never used as a
#: focus/distracted training label.
BREAK_PROTECTS_FROM_REMINDERS = True

#: Context that means "do not train the state model on this row".
NON_STATE_CONTEXTS: frozenset[str] = frozenset({"break"})


def normalise_context_type(value: object) -> str:
    """Coerce *value* to a known context type.

    Anything unrecognised — including ``None`` and empty strings — becomes
    ``unknown``. A value is never invented.
    """
    if value is None:
        return UNKNOWN
    text = str(value).strip().lower()
    return text if text in CONTEXT_TYPES else UNKNOWN


def normalise_goal_alignment(value: object) -> str:
    """Coerce *value* to a known goal alignment; unknown on anything else."""
    if value is None:
        return UNKNOWN
    text = str(value).strip().lower()
    return text if text in GOAL_ALIGNMENTS else UNKNOWN


def normalise_label_source(value: object) -> str:
    """Coerce *value* to a known label source; unknown on anything else."""
    if value is None:
        return UNKNOWN
    text = str(value).strip().lower()
    return text if text in LABEL_SOURCES else UNKNOWN


def is_state_supervision(context_type: object, label: object) -> bool:
    """Whether this row may supervise the focus/distracted state model.

    Returns ``False`` for a deliberate break and for the ambiguous ``mixed``
    label: neither is evidence about focus, and the spec forbids silently
    rewriting them into one.
    """
    if normalise_context_type(context_type) in NON_STATE_CONTEXTS:
        return False
    return str(label).strip().lower() in {"focus", "distracted"}


def protects_from_reminders(context_type: object) -> bool:
    """Whether this context should suppress automated reminders.

    A deliberate break is rest, not procrastination; interrupting it is the
    unwanted behaviour this project exists to avoid.
    """
    return normalise_context_type(context_type) == "break" and BREAK_PROTECTS_FROM_REMINDERS


__all__ = [
    "BREAK_PROTECTS_FROM_REMINDERS",
    "CONTEXT_TYPES",
    "GOAL_ALIGNMENTS",
    "LABEL_SOURCES",
    "NON_STATE_CONTEXTS",
    "TASK_TYPES",
    "UNKNOWN",
    "ContextType",
    "GoalAlignment",
    "HelpfulnessRating",
    "IntrusivenessRating",
    "LabelSource",
    "StateLabel",
    "is_state_supervision",
    "normalise_context_type",
    "normalise_goal_alignment",
    "normalise_label_source",
    "protects_from_reminders",
]
