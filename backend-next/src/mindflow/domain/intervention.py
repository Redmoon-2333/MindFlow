"""Intervention domain models — pure data, zero framework dependencies.

Defines the Intervention dataclass and supporting enums used throughout
the intervention engine (Wave 7).  No pydantic, no SQLAlchemy, no I/O.

Design constraints:
  - Frozen dataclasses (matching domain/events.py, domain/procrastination.py).
  - All message text is Chinese and never contains forbidden terms
    ("诊断/治疗/患者/处方" per NF-S7).
  - Intensity→message-template mapping is pure data — the intervention
    service applies it at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final, Literal

InterventionType = Literal[
    "task_breakdown",
    "nudge",
    "environment_optimization",
    "smart_prioritization",
]


class InterventionIntensity(StrEnum):
    """Three escalation levels for intervention tone and urgency.

    Members are lower-case strings matching the ``intensity`` parameter
    passed to ``InterventionService.maybe_intervene()``.
    """

    GENTLE = "gentle"
    STANDARD = "standard"
    STRICT = "strict"


class InterventionResponse(StrEnum):
    """User response to an intervention, stored in the DB log.

    ACCEPTED: User followed the suggestion.
    IGNORED: User dismissed without acting (timeout / close).
    DISMISSED: User explicitly rejected.
    """

    ACCEPTED = "accepted"
    IGNORED = "ignored"
    DISMISSED = "dismissed"


# ── Intensity -> (title_template, body_template) mapping ────────────────
# These are applied at runtime by the intervention service.
# {type_label} and {detail} are placeholders for runtime data.

_INTENSITY_GENTLE_TITLE: Final[str] = "小提示：{type_label}"
_INTENSITY_GENTLE_BODY: Final[str] = "注意到你目前{detail}，或许可以试试换个方式～"

_INTENSITY_STANDARD_TITLE: Final[str] = "来自 MindFlow 的提醒"
_INTENSITY_STANDARD_BODY: Final[str] = "检测到{detail}。建议尝试以下方法：{suggestion}"

_INTENSITY_STRICT_TITLE: Final[str] = "专注提醒"
_INTENSITY_STRICT_BODY: Final[str] = (
    "当前{detail}。请考虑调整策略：{suggestion}。持续注意对专注力的影响。"
)

INTENSITY_TEMPLATES: Final[dict[InterventionIntensity, tuple[str, str]]] = {
    InterventionIntensity.GENTLE: (_INTENSITY_GENTLE_TITLE, _INTENSITY_GENTLE_BODY),
    InterventionIntensity.STANDARD: (_INTENSITY_STANDARD_TITLE, _INTENSITY_STANDARD_BODY),
    InterventionIntensity.STRICT: (_INTENSITY_STRICT_TITLE, _INTENSITY_STRICT_BODY),
}

# ── Type → Chinese labels for template rendering ────────────────────────

INTERVENTION_TYPE_LABELS: Final[dict[str, str]] = {
    "task_breakdown": "任务分解",
    "nudge": "行动提示",
    "environment_optimization": "环境优化",
    "smart_prioritization": "优先级建议",
}


@dataclass(frozen=True)
class Intervention:
    """An immutable intervention recommendation.

    Attributes:
        id: UUIDv7 string.
        user_id: User identifier.
        intervention_type: One of the four intervention types.
        cbt_technique: The CBT technique that informed this intervention,
            or None for generic nudges.
        title: Notification title (Chinese, NF-S7 compliant).
        message: Notification body (Chinese, NF-S7 compliant).
        dismissible: Whether the user may dismiss this intervention.
            Always True for non-critical interventions.
        created_at: When this intervention was created (timezone-aware UTC).
    """

    id: str
    user_id: int
    intervention_type: InterventionType
    cbt_technique: str | None
    title: str
    message: str
    dismissible: bool
    created_at: datetime
